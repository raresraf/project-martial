// Codeforces
// 266B - Queue at the school

#include <algorithm>
#include <iostream>
using namespace std;

int main() {
  int numKids, numSeconds;
  cin >> numKids >> numSeconds;

  string queue;
  cin >> queue;

  for (unsigned int i = 0; i < numSeconds; i++) {
    for (unsigned int j = 0; j < queue.size() - 1; j++) {
      if (queue[j] == 'B' && queue[j + 1] == 'G') {
        swap(queue[j], queue[j + 1]);
        j++;
      }
    }
  }

  cout << queue;

  return 0;
}
