import torch

from data_processing.data_preprocessing import PredictionTransform
from ssd.utils import Timer
from ssd.utils import box_utils


class Predictor:
    """
    Klasa służąca do obsługi sieci SSD i zwracania rezultatów detekcji
    """

    def __init__(self, net, size, mean=0.0, std=1.0, nms_method=None,
                 iou_threshold=0.45, filter_threshold=0.01, candidate_size=200, sigma=0.5, device=None):
        """
        :param net: obiekt klasy SSD
        :param size: rozmiar obrazu będący długością boku kwadratu
        :param mean: średnia
        :param std: odchylenie standardowe
        :param nms_method: implementacja metody Non-Maximum Suppression
        :param iou_threshold: próg dla metody Intersection over Union
        :param filter_threshold: próg prawdopodobieństwa
        :param candidate_size: liczba kandydatów detekcji
        :param sigma: sigma
        :param device: GPU
        """
        self.net = net
        self.transform = PredictionTransform(size, mean, std)
        self.iou_threshold = iou_threshold
        self.filter_threshold = filter_threshold
        self.candidate_size = candidate_size
        self.nms_method = nms_method

        self.sigma = sigma
        if device:
            self.device = device
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.net.to(self.device)
        self.net.eval()

        self.timer = Timer()

    def predict(self, image, top_k=-1, prob_threshold=None):
        """
        Metoda zwraca wynik detekcji sieci SSD
        :param image: obraz w postaci array'a o kształcie (row, col, 3)
        :param top_k: parametr top K metody Non-Maximum Suppression
        :param prob_threshold: minimalny próg prawdopodobieństwa, który musi spełnić obiekt, aby zostać zwrócony
        :return: krotka 3 tensorów: boxes, labels, probs o kształtach (n, 4), (n, 1), (n, 1)
        """
        cpu_device = torch.device("cpu")
        height, width, _ = image.shape
        image = self.transform(image)
        images = image.unsqueeze(0)
        images = images.to(self.device)
        with torch.no_grad():
            self.timer.start()
            scores, boxes = self.net.forward(images)
        boxes = boxes[0]
        scores = scores[0]
        if not prob_threshold:
            prob_threshold = self.filter_threshold
        # this version of nms is slower on GPU, so we move data to CPU.
        boxes = boxes.to(cpu_device)
        scores = scores.to(cpu_device)
        picked_box_probs = []
        picked_labels = []
        for class_index in range(1, scores.size(1)):
            probs = scores[:, class_index]
            mask = probs > prob_threshold
            probs = probs[mask]
            if probs.size(0) == 0:
                continue
            subset_boxes = boxes[mask, :]
            box_probs = torch.cat([subset_boxes, probs.reshape(-1, 1)], dim=1)
            box_probs = box_utils.nms(box_probs, self.nms_method,
                                      score_threshold=prob_threshold,
                                      iou_threshold=self.iou_threshold,
                                      sigma=self.sigma,
                                      top_k=top_k,
                                      candidate_size=self.candidate_size)
            picked_box_probs.append(box_probs)
            picked_labels.extend([class_index] * box_probs.size(0))
        if not picked_box_probs:
            return torch.tensor([]), torch.tensor([]), torch.tensor([])
        picked_box_probs = torch.cat(picked_box_probs)
        picked_box_probs[:, 0] *= width
        picked_box_probs[:, 1] *= height
        picked_box_probs[:, 2] *= width
        picked_box_probs[:, 3] *= height
        return picked_box_probs[:, :4], torch.tensor(picked_labels), picked_box_probs[:, 4]
