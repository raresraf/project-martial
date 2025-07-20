#include <iostream>
#include <string>
#include <vector>

using namespace std;

int main(void) {
  int n, t;
  string queue, result;
  cin >> n >> t;
  cin >> queue;
  result = queue;

  for (int i = 0; i < t; i++) {
    for (int i = 0; i < queue.size() - 1; i++) {
      if (queue.at(i) == 'B' && queue.at(i + 1) == 'G') {
        result.at(i) = queue.at(i + 1);
        result.at(i + 1) = queue.at(i);
      }
    }

    queue = result;
  }

  cout << result << endl;
  return 0;
}