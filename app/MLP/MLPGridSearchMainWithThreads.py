import multiprocessing
import os
import shutil
import queue
from multiprocessing import Process
from app.MLP.MLPConfig import MLPConfig
from app.Runner import Runner
from app.Metrics import Metrics

finalLr = [0.002, 0.005, 0.01]
finalBatchsize = [1024, 1536, 2048]
finalHiddenLayerSize = [128, 256, 512]
finalNumberOfHiddenLayers = [2, 4, 6]

if os.path.exists("app/MLP/runs"): # clearing runs directory so that tensorboard shows recent statistics (tensorboard might still show cached data)
    shutil.rmtree("app/MLP/runs")

if os.path.isfile(r"../../app/MLP/best_model.pt"):
    os.remove(r"../../app/MLP/best_model.pt") # removing best model just in case

def trainOneModel(MLPConfig: MLPConfig):

    best_metric: Metrics | None = None
    runner = Runner(config=MLPConfig)

    trainer, test_loader, valMetrics = runner.startModel()

    test_acc, _, wrong_images = trainer.evaluate(test_loader)

    print(f"Final Epoch MLP Test Accuracy: {test_acc:.2f}%")

    valMetrics.final_best_val = valMetrics.getBestMetricElement(MLPConfig.patience).acc
    valMetrics.final_best_test = test_acc
    valMetrics.wrong_test_images = wrong_images

    if best_metric == None:
        best_metric: Metrics = valMetrics

    best_metric = valMetrics.checkBestMetric(best_metric=best_metric, config=MLPConfig)

    with open(rf'/Users/marcelanker/PycharmProjects/MINST_pytorch/app/MLP/results/MLP_lr{MLPConfig.learning_rate}_batchSize_{MLPConfig.batchsize}_hidden_{MLPConfig.hidden_layer_size}_layers_{MLPConfig.number_of_hidden_layers}.result', 'w') as file:
        file.write(f"Final best validation accuracy: {best_metric.final_best_val:.2f}% | "
          f"Final best test accuracy: {best_metric.final_best_test:.2f}% | "
          f"Final epoch mean time: {best_metric.final_epoch_mean_time:.2f}")




def worker(task_queue):
    while True:
        try:
            func, args = task_queue.get(timeout=1)
        except queue.Empty:
            print(f"{Process.name.__str__()} hat nix mehr zu tun.")
            break
        func(*args)



if __name__ == "__main__":

    manager = multiprocessing.Manager()
    best_metrics = manager.dict()
    lock = manager.Lock()

    tasks = []
    for lr in finalLr:
        for batchsize in finalBatchsize:
            for hidden_layer_size in finalHiddenLayerSize:
                for number_of_hidden_layers in finalNumberOfHiddenLayers:
                    config = MLPConfig(batchsize=batchsize, lr=lr, hidden_layer_size=hidden_layer_size, number_of_hidden_layers=number_of_hidden_layers, epochs=250, patience=7)
                    tasks.append((trainOneModel, {config: MLPConfig}))

    task_queue = manager.Queue()

    for aufgabe in tasks:
        task_queue.put(aufgabe)

    processes = []

    for i in range(5):
        p = Process(target=worker,args={task_queue}, name=f"Worker-{i + 1}")
        print('starting process')
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


