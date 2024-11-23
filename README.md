# jarvis.cpp
**jarvis.cpp** is a minimalistic and pure C++ implementation of an AI you can chat with using your voice,
similar to J.A.R.V.I.S from Iron Man.

## Install and Chat with jarvis.cpp.
```
git clone https://github.com/iangitonga/jarvis.cpp
cd jarvis.cpp/
g++ -std=c++17 -I. -O3 -fopenmp jarvis.cpp -o jarvis
./jarvis
```

If you have an Intel CPU that supports AVX and f16c compile with the following
 command to achieve higher performance:

```
g++ -std=c++17 -I. -O3 -fopenmp -mavx -mf16c jarvis.cpp -o jarvis
```

To see all the available options, run
```
./jarvis --help
```

