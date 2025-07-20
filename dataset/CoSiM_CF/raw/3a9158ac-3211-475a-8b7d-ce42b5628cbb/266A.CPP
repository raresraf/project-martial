#include <algorithm>
#include <cstdio>
#include <iostream>
using namespace std;

int main() {
  int cs, count = 0;
  string inp;
  char ch, last_ch;
  while (cin >> cs) {
    count = 0;
    cin >> inp;
    for (int i = 0; i < inp.size(); i++) {
      if (i == 0) {
        last_ch = inp[i];
        continue;
      }
      ch = inp[i];
      if (ch == last_ch) {
        count++;
      }
      last_ch = ch;
    }
    cout << count << endl;
  }
}
