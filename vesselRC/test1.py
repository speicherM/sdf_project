import logging
import time
import datetime
print(str(datetime.datetime.now()))
logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)
# 建立一个filehandler来把日志记录在文件里，级别为debug以上
fh = logging.FileHandler("log.log")
fh.setLevel(logging.DEBUG)
# 建立一个streamhandler来把日志打在CMD窗口上，级别为error以上
ch = logging.StreamHandler()
ch.setLevel(logging.ERROR)
# 设置日志格式
formatter = logging.Formatter("%(asctime)s - %(name)s - %(lineno)s %(message)s",datefmt="%Y-%m-%d %H:%M:%S")
ch.setFormatter(formatter)
fh.setFormatter(formatter)
#将相应的handler添加在logger对象中
logger.addHandler(ch)
logger.addHandler(fh)
# 开始打日志
logger.debug("debug message")
logger.info("info message")
logger.warning("warning message")
logger.error("error message")
logger.critical("critical message")

# import pprint
# stuff = ['spam', 'eggs', 'lumberjack', 'knights', 'ni']
# stuff.insert(0, stuff[:])
# #class pprint.PrettyPrinter(indent=1, width=80, depth=None, stream=None, *, compact=False)
# pp = pprint.PrettyPrinter(indent=4)
# pp.pprint(stuff)
# '''
# [   ['spam', 'eggs', 'lumberjack', 'knights', 'ni'],
#     'spam',
#     'eggs',
#     'lumberjack',
#     'knights',
#     'ni']

# '''