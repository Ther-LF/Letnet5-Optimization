### 安装基本编译环境
```
sudo apt-get install autoconf automake libtool
```
### 安装libunwind
```
cd libunwind-1.6.2
./configure
make
sudo make install
```
### 安装googleperf
```
cd gperftools-2.15
./autogen.sh
./configure
make
sudo make install
```



### googleperf 使用
以目标代码文件 test.cpp 为例
```
g++ test.cpp -o test -lprofiler -lunwind -ltcmalloc -lpthread -I./gperftools-2.15/build/include -L ./gperftools-2.15/build/lib/
# sudo 执行可执行文件test
sudo ./test
```
执行后生成的文件名可在test.cpp中的 ProfilerStart 的参数中设置(以test.prof为例)

使用pprof生成性能分析报告
```
./gperftools-2.15/build/bin/pprof ./test test.prof -pdf > test.pdf
```

