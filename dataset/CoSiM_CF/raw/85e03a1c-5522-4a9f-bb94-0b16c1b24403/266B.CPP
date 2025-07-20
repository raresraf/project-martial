#include <iostream>
#include <string.h>
using namespace std;
int main() {
  string queue;
  int numberstudents(0), time(0), iter(0);
  cin >> numberstudents >> time;
  cin >> queue;
  while (time--) {
    for (iter = 0; iter < numberstudents; iter++) {
      if (queue[iter] == 'B' && queue[iter + 1] == 'G') {
        queue[iter] = 'G';
        queue[iter + 1] = 'B';
        ++iter;
      }
    }
  }
  cout << queue;
  return 0;
}
