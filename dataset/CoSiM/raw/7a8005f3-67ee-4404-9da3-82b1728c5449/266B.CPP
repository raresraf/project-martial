#include <iostream>
#include <string>

using namespace std;

int main() {
  int numOfStu, time;
  string queue;

  cin >> numOfStu >> time;
  cin >> queue;

  for (int j = 0; j < time; j++) {
    for (int i = 0; i < queue.size() - 1; i++) {
      if ((queue[i] == 'B') && (queue[i + 1] == 'G')) {
        queue[i] = 'G';
        queue[i + 1] = 'B';
        i++;
      }
      // cout << i << endl;
    }
  }

  cout << queue << endl;

  return 0;
}
