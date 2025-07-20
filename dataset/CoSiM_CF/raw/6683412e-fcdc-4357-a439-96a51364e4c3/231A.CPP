#include <iostream>

using namespace std;

int main() {

  int n, num, count;
  cin >> n;
  int cap[n][3];
  count = 0;
  for (int i = 0; i < n; i++) {

    for (int j = 0; j < 3; j++) {

      cin >> cap[i][j];
    }
  }
  for (int i = 0; i < n; i++) {
    num = cap[i][0] + cap[i][1] + cap[i][2];
    if (num >= 2)
      count++;
  }
  cout << count;
  return 0;
}