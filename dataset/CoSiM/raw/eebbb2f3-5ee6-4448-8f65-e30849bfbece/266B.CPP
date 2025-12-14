#include <iostream>
using namespace std;

int main(int argc, char *argv[]) {
  int n, t;
  string queue;
  cin >> n >> t;
  cin >> queue;
  if (n == 1) {
    cout << queue << endl;
    return 0;
  }
  while (t--) {
    for (int i = 0; i < queue.length() - 1; i++) {
      if (queue.at(i) == 'B' && queue.at(i + 1) == 'G') {
        queue[i] = 'G';
        queue[i + 1] = 'B';
        i++;
      }
    }
  }
  cout << queue << endl;
  return 0;
}
