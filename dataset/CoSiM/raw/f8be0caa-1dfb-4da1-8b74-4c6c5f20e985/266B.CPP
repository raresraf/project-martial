#include <iostream>

using namespace std;

void swap(int &x, int &y) {
  int tmp;
  tmp = x;
  x = y;
  y = tmp;
}

int main() {
  int n, t;
  char queue[50];
  cin >> n >> t >> queue;

  for (int i = 0; i < t; i++) {
    int j = 0;
    while (queue[j + 1] != '\0') {
      if (queue[j] == 'B' && queue[j + 1] == 'G') {
        swap(queue[j], queue[j + 1]);
        j++;
        if (queue[j + 1] == '\0')
          break;
      }
      j++;
    }
  }
  cout << queue;
  return 0;
}