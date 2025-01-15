#include <algorithm>
#include <iostream>

using namespace std;

int main() {
  int n, t;
  cin >> n >> t;
  string childrenQueue;
  cin >> childrenQueue;
  bool replaced = false;

  for (int i = 0; i < t; i++) {
    for (int j = 0, k = 1; k < childrenQueue.size(); j++, k++) {
      if (childrenQueue[j] == 'B' && childrenQueue[k] == 'G' && !replaced) {
        swap(childrenQueue[j], childrenQueue[k]);
        replaced = true;
        continue;
      }
      if (replaced)
        replaced = false;
    }
    if (replaced)
      replaced = false;
  }

  cout << childrenQueue;

  return 0;
}