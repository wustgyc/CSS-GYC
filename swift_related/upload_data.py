
from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, wait, ALL_COMPLETED, as_completed
import os
from datetime import datetime
import random
IMG_FILE_PATH="/tfproject/DSTH-TF-YFLiu/datasets/Imagenet_png_train_1million_32/train_32x32"
auth_token = "AUTH_tk1f087a802596473fa5a249f4c65aaab3"


def thread_func(file_name):
    host=random.randint(13,14)
    swift_cmd = "curl -X PUT -T " + os.path.join(IMG_FILE_PATH,file_name) + " -H 'X-Auth-Token: " + auth_token + "' http://192.168.0."+str(host)+":8080/v1/AUTH_test/img_data/"

    os.system(swift_cmd)


def pool_func(file_list):
    # 进程函数
    executor = ThreadPoolExecutor(max_workers=thread_num)  # 定义线程池，设置最大线程数量
    ret_list=[]
    for file_name in file_list:
        ret_list.append(executor.submit(thread_func,file_name)) # 将线程添加到线程池

    for ret in as_completed(ret_list):
        ret.result()

    executor.shutdown(wait=True)

if __name__ == "__main__":
    thread_num = 20  # 线程最大数量
    pool_num = 8  # 进程数量

    pool = ProcessPoolExecutor(max_workers=pool_num)  # 定义进程池

    file_list = os.listdir(IMG_FILE_PATH)
    NUMBER = len(file_list)
    batch_size = NUMBER // (pool_num*2)
    batch_size+=1
    begin=datetime.now()
    ret_list=[]
    for i in range(pool_num*2):
        job_list = file_list[batch_size*i: (i+1)*batch_size]
        ret_list.append(pool.submit(pool_func, job_list))

    for ret in as_completed(ret_list):
        ret.result()



    end = datetime.now()
    total_times=(end-begin).total_seconds()
    print("total times:",total_times)

    pool.shutdown(wait=True)