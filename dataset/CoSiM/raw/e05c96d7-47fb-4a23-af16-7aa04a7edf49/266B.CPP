#include <bits/stdc++.h>
using namespace std;

int main(void) {
  int childrenSize, time;
  string queue;

  cin >> childrenSize >> time >> queue;

  while (time--) {
    for (auto i = queue.begin(); i != queue.end(); ++i) {
      if (*i == 'B' && *(i + 1) == 'G') {
        *i = 'G';
        *(i + 1) = 'B';
        ++i;
      }
    }
  }

  cout << queue;
  return 0;
}
