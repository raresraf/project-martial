#include <iostream>

using namespace std;

int main() {
  int n, t;
  string queue;
  cin >> n >> t >> queue;
  while (t--) {
    for (int i = 1; i < n; i++) {
      if (queue[i - 1] == 'B' && queue[i] == 'G') {
        queue[i] = 'B';
        queue[i - 1] = 'G';
        i++;
      }
    }
  }
  cout << queue << endl;
  return 0;
}