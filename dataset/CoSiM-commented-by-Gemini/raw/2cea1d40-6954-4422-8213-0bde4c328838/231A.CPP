#include <iostream>
using namespace std;
int main() {
  int n, count = 0, answ = 0, data[1000][5];
  cin >> n;
  for (int i = 0; i < n; i++) {
    count = 0;
    for (int j = 0; j < 3; j++) {
      cin >> data[i][j];
      if (data[i][j] == 1)
        count++;
    }
    if (count >= 2)
      answ++;
  }
  cout << answ;
  return 0;
}
