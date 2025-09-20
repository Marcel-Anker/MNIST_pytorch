import multiprocessing
import os
import shutil
import queue
from multiprocessing import Process
from app.CNN.CNNConfig import CNNConfig
from app.Runner import Runner
from app.Metrics import Metrics

finalLr = [0.002, 0.005, 0.01]
finalBatchsize = [1024, 1536, 2048]
finalNumberConvLayers = [1, 2, 3]
finalKernelSize = [2, 3, 4]
finalConvStride = [1, 2, 3]

if os.path.exists("app/CNN/runs"): # clearing runs directory so that tensorboard shows recent statistics (tensorboard might still show cached data)
    shutil.rmtree("app/CNN/runs")

if os.path.isfile(r"../../app/CNN/best_model.pt"):
    os.remove(r"../../app/CNN/best_model.pt") # removing best model just in case

def trainOneModel(CNNConfig: CNNConfig):

    best_metric: Metrics | None = None
    runner = Runner(config=CNNConfig)

    trainer, test_loader, valMetrics = runner.startModel()

    test_acc, _, wrong_images = trainer.evaluate(test_loader)

    print(f"Final Epoch CNN Test Accuracy: {test_acc:.2f}%")

    valMetrics.final_best_val = valMetrics.getBestMetricElement(CNNConfig.patience).acc
    valMetrics.final_best_test = test_acc
    valMetrics.wrong_test_images = wrong_images

    if best_metric == None:
        best_metric: Metrics = valMetrics

    best_metric = valMetrics.checkBestMetric(best_metric=best_metric, config=CNNConfig)

    with open(rf'/Users/marcelanker/PycharmProjects/MINST_pytorch/app/CNN/results/CNN_lr{CNNConfig.learning_rate}_batchSize_{CNNConfig.batchsize}_number_conv_{CNNConfig.number_conv_layers}_kernel_size_{CNNConfig.kernel_size}_stride_{CNNConfig.conv_stride}.result', 'w') as file:
        file.write(f"Final best validation accuracy: {best_metric.final_best_val:.2f}% | "
          f"Final best test accuracy: {best_metric.final_best_test:.2f}% | "
          f"Final epoch mean time: {best_metric.final_epoch_mean_time:.2f}")




def worker(task_queue):
    while True:
        try:
            func, args = task_queue.get(timeout=1)
        except queue.Empty:
            print(f"{Process.name.__str__()} is waiting.")
            break
        func(*args)



if __name__ == "__main__":

    manager = multiprocessing.Manager()
    best_metrics = manager.dict()  # geteilt über alle Prozesse
    lock = manager.Lock()

    tasks = []
    for lr in finalLr:
        for batchsize in finalBatchsize:
            for number_conv_layers in finalNumberConvLayers:
                for kernel_size in finalKernelSize:
                    for conv_stride in finalConvStride:
                        config = CNNConfig(batchsize=batchsize, lr=lr, number_conv_layers=number_conv_layers,
                                           kernel_size=kernel_size, out_channels=4, conv_stride=conv_stride, epochs=250,
                                           patience=7)
                        tasks.append((trainOneModel, {config: CNNConfig}))

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


