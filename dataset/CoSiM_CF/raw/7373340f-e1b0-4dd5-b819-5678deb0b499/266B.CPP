#include <bits/stdc++.h>
#include <iostream>
#include <string>
//#include <vector>
//#include <math.h>

#define LOG(x) cout << (x) << endl;
#define ARR_LOG(x)                                                             \
  for (auto c : x) {                                                           \
    LOG(c)                                                                     \
  };
//#define ALL(x)      (x).begin(), (x).end()

using namespace std;

int main() {
  int n = 0, t = 0;
  string inpStr = "";
  cin >> n >> t >> inpStr;

  while (t != 0) {
    for (auto i = 0; i < n - 1; i++) {
      if (inpStr[i] == 'B' && inpStr[i + 1] == 'G') {
        inpStr[i] = inpStr[i] ^ inpStr[i + 1];
        inpStr[i + 1] = inpStr[i] ^ inpStr[i + 1];
        inpStr[i] = inpStr[i] ^ inpStr[i + 1];
        i++;
      }
    }
    t--;
  }

  LOG(inpStr)
  return 0;
}