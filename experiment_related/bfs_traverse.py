# coding: utf-8
# 默认hashcode长度为48
from __future__ import division
import logging

import h5py
import random
import multiprocessing
from neo4j.v1 import GraphDatabase, basic_auth
import datetime
import pylru
import argparse
import os
from caculate_object_memory_use import total_size
import numpy as np
import csv
import warnings
import time
from multiprocessing import Manager
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, wait, ALL_COMPLETED, as_completed
import os
import cv2
import subprocess

warnings.filterwarnings("ignore")

__PATH2__ = '/NewDisk/gyc/CSS-CACHE/data'
DOWN_LOAD_PATH = '/NewDisk/gyc/CSS-CACHE/experiment_related/download_data'
TMP_PATH = '/NewDisk/gyc/CSS-CACHE/experiment_related/tmp'
IMG_FILE_PATH = "/tfproject/DSTH-TF-YFLiu/datasets/Imagenet_png_train_1million_32/train_32x32"
CACHE_STORE_PATH = "/NewDisk/gyc/CSS-CACHE/experiment_related/cache_store_space"
Ihd = 2
cache_use = True
random.seed(0)
auth_token = "AUTH_tk1f087a802596473fa5a249f4c65aaab3"
neo4j_uri = "bolt://127.0.0.1:7687"
download_process_num = 4  # 不能再大了，再大负载受不了
download_thread_num = 16


def cache_ejected_callback(key, value):
    try:
        cmd = "rm -rf " + os.path.join(CACHE_STORE_PATH, key)
        os.system(cmd)
        return 0
    except Exception as e:
        print(e)
        return 1


def str_to_list(str):
    if len(str) == 0: return []
    assert str[-1] == ';'
    str = str[:-1]
    str.split(";", -1)
    return str.split(";", -1)


def list_to_str(list):
    ret = ""
    for row in list:
        ret += row + ';'
    return ret


def check_cache(cache_distance_item, shg_distance_item, radius):
    need_to_add_distance_item = ["", "", "", ""]
    assert len(cache_distance_item) == len(shg_distance_item)

    for i in range(int(radius / 2)):
        cache_relevant_data_set = set(str_to_list(cache_distance_item[i]))
        shg_relevant_data_set = set(str_to_list(shg_distance_item[i]))
        need_to_add_relevant_data_list = list(shg_relevant_data_set - cache_relevant_data_set)
        need_to_add_distance_item[i] = list_to_str(need_to_add_relevant_data_list)

    return need_to_add_distance_item


def traverse_SHG(hash, radius):
    SimiNodes_filename_str = []
    SimiNodes_hash_str = []

    with GraphDatabase.driver(neo4j_uri, auth=basic_auth("neo4j", "123456")) as cdriver:
        with cdriver.session() as session:
            cypher_cmd = "match(n:IHashNode {hashCode:'" + hash + "'}) call algo.bfs.stream('IHashNode', 'Simi', 'OUTGOING', id(n),{maxCost:" + str(
                radius + 1) + ", weightProperty:'dist'}) yield nodeIds unwind nodeIds as nodeId return algo.asNode(nodeId).file_list,algo.asNode(nodeId).hashCode"  # BFS的时候第一个结点会占1个dist，所以args.dist_thre+1

            print(cypher_cmd)
            try:
                ret = session.run(cypher_cmd)
            except Exception as e:
                print(e)
                return [], []

    for record in ret:
        str_record_0 = str(record[0]) + ';'
        str_record_1 = str(record[1])
        SimiNodes_filename_str.append(str_record_0)  # ['a;b;c;','d;g;','x;y;z;']
        SimiNodes_hash_str.append(str_record_1)

    print('SHG:', len(str_to_list(array_join(SimiNodes_filename_str))))
    return SimiNodes_filename_str, SimiNodes_hash_str


def download_process(download_job_list, mod):
    download_thread_pool = ThreadPoolExecutor(max_workers=download_thread_num)
    img_list = []
    thread_ret = []
    for job in download_job_list:
        thread_ret.append(download_thread_pool.submit(download_thread, job, mod))
    for ret in as_completed(thread_ret):
        img_list.append(ret.result())

    download_thread_pool.shutdown(wait=True)
    return img_list


def download_thread(job, mod):
    curr_name = job[0]
    dist = job[1]
    curr_name_path = job[2]
    proxy_host = str(random.randint(13, 14))
    swift_cmd = "curl -s -o " + os.path.join(CACHE_STORE_PATH, curr_name,
                                             dist,
                                             curr_name_path) + " -H 'X-Auth-Token: " + auth_token + "' http://192.168.0." + str(
        proxy_host) + ":8080/v1/AUTH_test/img_data/" + curr_name_path
    os.system(swift_cmd)
    if mod == "add":  # 新增的就别麻烦了，直接在这里面放到内存里好了
        img = cv2.imread(os.path.join(CACHE_STORE_PATH, curr_name, dist, curr_name_path))
        return img


def download_from_swift(curr_name, shg_distance_item, mod):
    if mod == "write":
        os.system("mkdir -p " + os.path.join(CACHE_STORE_PATH, curr_name))
    download_job_list = []
    for i, curr_relevant_data in enumerate(shg_distance_item):
        dist = str((i + 1) * 2)
        os.system("mkdir -p " + os.path.join(CACHE_STORE_PATH, curr_name, dist))
        for curr_name_path in str_to_list(curr_relevant_data):
            download_job_list.append((curr_name, dist, curr_name_path))

    batch_size = len(download_job_list) // (download_process_num * 3)
    batch_size += 1  # 怕除不尽
    process_ret = []
    for i in range(download_process_num * 3):
        process_ret.append(download_process_pool.submit(download_process,
                                                        download_job_list[i * batch_size:(i + 1) * batch_size],
                                                        mod))

    # 阻塞
    if mod == "add":
        img_list = []
        for ret in as_completed(process_ret):
            img_list.extend(ret.result())
        print('swift add:', len(img_list))
        return img_list

    else:
        for ret in as_completed(process_ret):
            ret.result()


def hammingDistBit(hashstr1, hashstr2):
    """Calculate the Hamming distance between two bit strings"""
    return bin(int(hashstr1, 2) ^ int(hashstr2, 2)).count('1')


def sort_relevant_to_array(file_name_list, hashcode_list, original_hash):
    distance_item = ["", "", "", ""]
    assert len(file_name_list) == len(hashcode_list)
    for i, row in enumerate(hashcode_list):
        if hammingDistBit(row, original_hash) <= 2:
            distance_item[0] += file_name_list[i]
        elif hammingDistBit(row, original_hash) <= 4:
            distance_item[1] += file_name_list[i]
        elif hammingDistBit(row, original_hash) <= 6:
            distance_item[2] += file_name_list[i]
        elif hammingDistBit(row, original_hash) <= 8:
            distance_item[3] += file_name_list[i]
        else:
            print("danger!")
    return distance_item


def array_join(array):
    ret = ""
    for row in array:
        ret += row
    return ret


def read_from_cache_to_memory(curr_name, radius):
    img_list = []
    for rd in range(2, radius + 1, 2):
        for root, dirs, files in os.walk(os.path.join(CACHE_STORE_PATH, curr_name, str(rd)), topdown=False):
            for name in files:
                img_list.append(cv2.imread(os.path.join(root, name)))
    print("read from cache:", len(img_list))
    return img_list


def analyse_query(curr_name, curr_hash, semi_radius=8):
    logger.debug("analyse_query:" + " " + curr_name + " " + curr_hash + " " + str(semi_radius))
    global point
    if curr_name in cache.table:
        assert curr_hash == cache[curr_name][0]

        # 如果要写入子进程则需声明共享变量
        cache_distance_item = cache[curr_name][1:]  # 这里仅仅是为了后面check用

        # shg校验与cache读入内存同时进行
        res_cache = consistency_process_pool.submit(read_from_cache_to_memory, curr_name, semi_radius)
        res_shg = consistency_process_pool.submit(traverse_SHG, curr_hash, semi_radius)

        # 等待进程完成
        shg_file_name_2d, shg_hash_1d = res_shg.result()
        img_list = res_cache.result()

        if curr_name not in str_to_list(array_join(shg_file_name_2d)):   #哈希给对了，但是文件名瞎几把输的清空
            return []
        # use img_list to do something

        # check
        shg_distance_item = sort_relevant_to_array(shg_file_name_2d, shg_hash_1d, curr_hash)
        need_to_add_item = check_cache(cache_distance_item, shg_distance_item, semi_radius)
        img_list.extend(download_from_swift(curr_name, need_to_add_item, "add"))  # add something to cache

        # update_cache 无论如何以新的为准
        cache[curr_name][0] = curr_hash
        cache[curr_name][1:] = shg_distance_item

        point += 1

        return img_list
    else:
        shg_file_name_2d, shg_hash_1d = traverse_SHG(curr_hash, semi_radius)
        if curr_name not in str_to_list(array_join(shg_file_name_2d)):
            return []
        # put into cache
        cache[curr_name] = ["", "", "", "", ""]
        cache[curr_name][0] = curr_hash
        shg_distance_item = sort_relevant_to_array(shg_file_name_2d, shg_hash_1d, curr_hash)
        cache[curr_name][1:] = shg_distance_item

        # download and read
        download_from_swift(curr_name, shg_distance_item, "write")  # 按hashcode下载到相应文件夹
        img_list = read_from_cache_to_memory(curr_name, semi_radius)

        # use img_list to do something
        return img_list


def insert(curr_name, curr_hash):        #改！！！！！！！ 判断match
    def insert_SHG():
        # 搜索所有结点，计算边
        file_with_same_hash = curr_name
        Simi_node = []
        for i, row in enumerate(hashcode):
            tra_hash = hashcode[i]
            dist = hammingDistBit(curr_hash, tra_hash)
            if dist <= Ihd:
                Simi_node.append((tra_hash, dist))
            if dist == 0:
                file_with_same_hash += ";" + name_path_list[i]
        with GraphDatabase.driver(neo4j_uri, auth=basic_auth("neo4j", "123456")) as c_driver:
            with c_driver.session() as session:
                cypher_cmd = "match (n:IHashNode{hashCode:'" + curr_hash + "'}) return count(n)"
                shg_ret=session.run(cypher_cmd)
                for row in shg_ret:
                    node_exist=row[0]
                if node_exist==0:
                    #加入点
                    cypher_cmd = "create (n:IHashNode{hashCode:'" + curr_hash + "',file_list:'" + file_with_same_hash + "'})"  # 已添加唯一性约束，直接create
                    try:
                        session.run(cypher_cmd)
                    except Exception as e:
                        print(e)
                        return 1

                    # 计算边
                    for i, si_node in enumerate(Simi_node):
                        si_hash = si_node[0]
                        si_dist = si_node[1]
                        print(si_dist)
                        # 加入正向边
                        cypher_cmd = "match (begin:IHashNode{hashCode:'" + curr_hash + "'}) match (end:IHashNode{hashCode:'" + si_hash + "'}) merge (begin)-[r:Simi{dist:" + str(
                            si_dist) + "}]->(end)"
                        try:
                            session.run(cypher_cmd)
                        except Exception as e:
                            print(e)
                            return 1

                        # 加入反向边
                        cypher_cmd = "match (begin:IHashNode{hashCode:'" + curr_hash + "'}) match (end:IHashNode{hashCode:'" + si_hash + "'}) merge (end)-[r:Simi{dist:" + str(
                            si_dist) + "}]->(begin)"
                        try:
                            session.run(cypher_cmd)
                        except Exception as e:
                            print(e)
                            return 1
                else:
                    cypher_cmd = "match (n:IHashNode{hashCode:'" + curr_hash + "'}) set n.file_list='" + file_with_same_hash + "'"
                    try:
                        session.run(cypher_cmd)
                    except Exception as e:
                        print(e)
                        return 1

        return 0

    def inset_swift():
        swift_cmd = "curl -s -X PUT -T " + os.path.join(IMG_FILE_PATH,
                                                        curr_name) + " -H 'X-Auth-Token: " + auth_token + "' http://192.168.0.14:8080/v1/AUTH_test/img_data/"
        os.system(swift_cmd)
        return 0

    logger.debug('insert:' + " " + curr_hash + " " + curr_name)

    insert_pool = ThreadPoolExecutor(2)

    ret_1 = insert_SHG()
    ret_2 = inset_swift()
    if ret_1 == 0 and ret_2 == 0:
        print(curr_name, "insert success")
        return 0
    else:
        print("insert fail")
        return 1


def delete(curr_name, curr_hash):
    def delete_SHG():
        with GraphDatabase.driver(neo4j_uri, auth=basic_auth("neo4j", "123456")) as c_driver:
            with c_driver.session() as session:
                cypher_cmd = "match (n:IHashNode{hashCode:'" + curr_hash + "'}) return n.file_list"
                shg_ret = session.run(cypher_cmd)
                file_str="abab"
                for row in shg_ret:
                    file_str = row[0]
                file_str = file_str.replace(curr_name + ';', '')
                file_str = file_str.replace(';' + curr_name, '')
                file_str = file_str.replace(curr_name, '')
                cypher_cmd = "match (n:IHashNode{hashCode:'" + curr_hash + "'}) set n.file_list='" + file_str + "'"
                try:
                    session.run(cypher_cmd)
                except Exception as e:
                    print(e)

                if file_str=="":   #最后一个
                    # 删边
                    cypher_cmd = "match (n:IHashNode{hashCode:'" + curr_hash + "'})-[r]-(m) delete r"
                    try:
                        session.run(cypher_cmd)
                    except Exception as e:
                        print(e)

                    # 删点
                    cypher_cmd = "match (n:IHashNode{hashCode:'" + curr_hash + "'}) delete n"
                    try:
                        session.run(cypher_cmd)
                    except Exception as e:
                        print(e)
                        return 1
        return 0

    def delete_swift():
        host = random.randint(13, 14)
        swift_cmd = "curl -s -X DELETE -H 'X-Auth-Token: " + auth_token + "' http://192.168.0." + str(
            host) + ":8080/v1/AUTH_test/img_data/" + curr_name
        curl_ret = "bad curl"
        limit = 0

        while curl_ret != "" and limit <= 10 and "The resource could not be found" not in curl_ret:  # 多线程下可能curl失败，尝试重做
            curl_ret = os.popen(swift_cmd).read().strip()
            limit += 1
            if limit >= 2:
                print("curl_ret", curl_ret)
        if limit > 10:
            return 1
        else:
            return 0

    logger.debug('delete:' + " " + curr_hash + " " + curr_name)
    ret_1 = delete_SHG()
    ret_2 = delete_swift()
    if ret_1 == 0 and ret_2 == 0:
        print(curr_name, "delete success")
        return 0
    else:
        print("delete fail")
        return 1


def edit_file_name(old_name, new_name, curr_hash):
    def edit_SHG(old_name, new_name, curr_hash):
        with GraphDatabase.driver(neo4j_uri, auth=basic_auth("neo4j", "123456")) as c_driver:
            cypher_cmd = "match (n:IHashNode{hashCode:'" + curr_hash + "'}) return n.file_list"
            try:
                with c_driver.session() as session:
                    shg_ret = session.run(cypher_cmd)
                    for row in shg_ret:
                        file_str = row[0]
                    file_str = file_str.replace(old_name, new_name)
                    cypher_cmd = "match (n:IHashNode{hashCode:'" + curr_hash + "'}) set n.file_list='" + file_str + "'"
                    try:
                        session.run(cypher_cmd)
                    except Exception as e:
                        print(e)
                        return 1
            except Exception as e:
                print(e)
                return 1
        return 0

    def edit_swift(old_name, new_name):
        # 下载
        swift_cmd = "curl -s -o " + os.path.join(TMP_PATH,
                                                 new_name) + " -H 'X-Auth-Token: " + auth_token + "' http://192.168.0.14:8080/v1/AUTH_test/img_data/" + old_name
        try:
            os.system(swift_cmd)
        except Exception as e:
            print(e)
            return 1

        # 删除
        swift_cmd = "curl -s -X DELETE -H 'X-Auth-Token: " + auth_token + "' http://192.168.0.14:8080/v1/AUTH_test/img_data/" + old_name
        try:
            os.system(swift_cmd)
        except Exception as e:
            print(e)
            return 1

        # 重新上传
        swift_cmd = "curl -s -X PUT -T " + os.path.join(TMP_PATH,
                                                        new_name) + " -H 'X-Auth-Token: " + auth_token + "' http://192.168.0.14:8080/v1/AUTH_test/img_data/"
        try:
            os.system(swift_cmd)
        except Exception as e:
            print(e)
            return 1
        return 0

    logger.debug('edit_file_name:' + " " + curr_hash + " " + old_name + " to " + new_name)
    ret_1 = edit_SHG(old_name, new_name, curr_hash)
    ret_2 = edit_swift(old_name, new_name)

    if ret_1 == 0 and ret_2 == 0:
        print("edit_1 success!")
        return 0
    else:
        print("edit_1 faile")
        return 1


def edit_file_content(old_name, old_hash, new_name, new_hash):  #改,driver变了
    ret_1 = delete(old_name, old_hash)
    ret_2 = insert(new_name, new_hash)

    logger.debug('edit_file_content:' + "(" + old_hash + "," + old_name + ")", " to ",
                 "(" + new_hash + "," + new_name + ")")
    if ret_1 == 0 and ret_2 == 0:
        print("edit_2 sucess")
        return 0
    else:
        print("edit_2 fail")
        return 1


# 临时用
def delete_fun(job_list_name,job_list_hash):
    del_thread_pool = ThreadPoolExecutor(max_workers=download_thread_num)
    thread_ret = []

    for i,row in enumerate(job_list_name):
        thread_ret.append(del_thread_pool.submit(delete, job_list_name[i], job_list_hash[i]))

    for ret in as_completed(thread_ret):
        ret.result()

    del_thread_pool.shutdown(wait=True)
    return 0


if __name__ == "__main__":
    # parse parameter
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--cache_size', type=int, default=6000)
    parser.add_argument('--max_query_times', type=int, default=6000)
    parser.add_argument('--query_range', type=int, default=6000)

    args = parser.parse_args()
    print("")
    print("cache size", args.cache_size)
    print("max query times", args.max_query_times)
    print("query range", args.query_range)
    print("")

    # log,记录增删改查操作
    logging.basicConfig(filename=os.path.join(os.getcwd(), 'operate.log'), level=logging.DEBUG)
    logger = logging.getLogger('admin operate')
    # init cache
    cache = pylru.lrucache(10, cache_ejected_callback)
    cache.clear()
    if os.path.exists('cache.npy'):
        cache.table = np.load('cache.npy').item()

    # init dataset
    with h5py.File(os.path.join(__PATH2__, 'ImageNet_1million_png_predicthashstr_train.hdf5'), 'r') as f:
        hashcode = f['predicthashstr'].value
        hashcode = hashcode.astype(np.str)

    with open(os.path.join(__PATH2__, "ImageNet_path_traverse_file.txt"), 'r') as f:
        name_path_list = f.readlines()
    for i, line in enumerate(name_path_list):  # 去掉\n
        name_path_list[i] = line[:-1]
        name_path_list[i] = name_path_list[i].split('/', -1)[-1]
    name_path_list = name_path_list[:len(hashcode)]  # 统一长度

    # 划分集合
    hashcode_new = hashcode[1000000:]  # 新增集
    hashcode = hashcode[:1000000]  # 原始集
    name_path_list_new = name_path_list[1000000:]  # 新增集
    name_path_list = name_path_list[:1000000]  # 原始集

    # init parameter
    point_rate_avg = []
    time_avg = []
    mem_use_avg = []
    query_times_list = []
    overload_divide = 10
    point = 0
    # 负载阶段划分
    assert args.max_query_times % overload_divide == 0
    for i in range(1, overload_divide + 1):
        point_rate_avg.append([])
        time_avg.append([])
        mem_use_avg.append([])
        query_times_list.append(i * args.max_query_times // overload_divide)

    # init neo4j


    # init process_pool
    consistency_process_pool = ProcessPoolExecutor(max_workers=2)  # shg与swift并行的进程池
    download_process_pool = ProcessPoolExecutor(max_workers=download_process_num)  # 下载专用进程池

    begin = datetime.datetime.now()
    # for i,row in enumerate(hashcode_new):
    #     cmd="match (n:IHashNode{hashCode:'"+row+"'}) return count(n)"
    #     with GraphDatabase.driver(neo4j_uri, auth=basic_auth("neo4j", "123456")) as c_driver:
    #         with c_driver.session() as session:
    #             shg_ret=session.run(cmd)
    #         for row in shg_ret:
    #             exist=row[0]
    #         if exist==0:
    #             print(hashcode_new[i],name_path_list_new[i],i)
    #             exit(0)


    # print(name_path_list_new[0],hashcode_new[0])
    insert(name_path_list_new[2], hashcode_new[2])
    delete(name_path_list_new[2], hashcode_new[2])
    
    end = datetime.datetime.now()
    print("cost time:",(end-begin).total_seconds())
    # 开始计时
    # insert(name_path_list_new[0], hashcode_new[0])
    # begin=datetime.datetime.now()
    # analyse_query(name_path_list[0],hashcode[0])
    # end = datetime.datetime.now()
    # print("cost time:",(end-begin).total_seconds())



    # i = -1
    # for ans_num, query_times in enumerate(query_times_list):
    #     for i in range(i + 1, query_times):
    #         rand_id = random.randint(0, args.query_range - 1)
    #         curr_hash = hashcode[rand_id]
    #         curr_name = name_path_list[rand_id]
    #         query(curr_hash, curr_name)
    #
    #     subtime_end = datetime.datetime.now()
    #     mem_use = total_size(cache.table) / 1024 / 1024
    #     print("query times:", query_times)
    #     print("point rate:", 1.0 * point / query_times
    #     print("cost time:", (subtime_end - subtime_start).total_seconds()
    #     print("cache_memory used:", mem_use, "MB"
    #     point_rate_avg[ans_num].append(1.0 * point / query_times)
    #     time_avg[ans_num].append((subtime_end - subtime_start).total_seconds())
    #     mem_use_avg[ans_num].append(mem_use)
    #
    #     with open('./css_cache_experiment.csv', 'a') as f:
    #         csv_writer = csv.writer(f)
    #         csv_writer.writerow(
    #             [datetime.datetime.now(), Ihd, args.dist_thre, cache_use, query_times,
    #              args.query_range,
    #              np.mean(time_avg[ans_num]), np.std(time_avg[ans_num]), args.cache_size,
    #              np.mean(mem_use_avg[ans_num]),
    #              np.mean(point_rate_avg[ans_num])])
    #         f.flush()

    # insert(hashcode[2], name_path_list[2])
    # insert(hashcode[1], name_path_list[1])

    # delete(hashcode[0], name_path_list[0])

    # analyse_query(hashcode[0], name_path_list[0])
    # edit_file_name(hashcode[0], name_path_list[0],'papa.png')
    # get_from_swift([os.path.join(IMG_FILE_PATH,name_path_list[0])])
    # insert(hashcode[0], name_path_list[0])
    # get_from_swift([name_path_list[0]])
    # print(1111
    # for i in range(6):
    #     print(name_path_list[i]
    # get_from_swift([name_path_list[1]])
    # get_from_swift(['papa.png'])
    # file_name,hash_list=traverse_SHG(hashcode[0])
    # print(array_join(file_name).find(name_path_list[0])
    # print(array_join(file_name).find('papa.png')

    # 持久化cache
    np.save('cache.npy', cache.table)

    np.save('../data/hashcode.npy',hashcode)
    np.save('../data/hashcode_new.npy', hashcode_new)
    np.save('../data/name_path_list', name_path_list)
    np.save('../data/name_path_list_new', name_path_list_new)
    # 退出
    download_process_pool.shutdown(wait=True)
    consistency_process_pool.shutdown(wait=True)
