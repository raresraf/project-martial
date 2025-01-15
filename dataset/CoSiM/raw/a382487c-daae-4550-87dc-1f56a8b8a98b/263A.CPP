#include <iostream>
using namespace std;

int main() {
  // your code goes here
  int a[5][5];
  pair<int, int> cord;
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      cin >> a[i][j];
      if (a[i][j] == 1) {
        cord = {i, j};
      }
    }
  }
  cout << abs(2 - cord.first) + abs(2 - cord.second) << endl;
  return 0;
}