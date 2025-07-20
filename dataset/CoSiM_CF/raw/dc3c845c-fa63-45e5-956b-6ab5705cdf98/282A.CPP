#include <algorithm>
#include <cstdio>
#include <iostream>
using namespace std;

int main() {
  int cs, x = 0;
  string inp;
  while (cin >> cs) {
    x = 0;
    for (int i = 0; i < cs; i++) {
      cin >> inp;
      if (inp == "X++" || inp == "++X") {
        ++x;
      }
      if (inp == "X--" || inp == "--X") {
        --x;
      }
    }
    cout << x << endl;
  }
}
