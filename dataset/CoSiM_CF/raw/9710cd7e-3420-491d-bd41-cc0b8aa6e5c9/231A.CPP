#include <iostream>
using namespace std;
int main() {
  int n, i;
  cin >> n;
  int q[1000][3];
  for (i = 0; i < n; i++) {
    cin >> q[i][0] >> q[i][1] >> q[i][2];
  }
  int count = 0;
  int temp;
  for (i = 0; i < n; i++) {
    temp = q[i][0] + q[i][1] + q[i][2];
    if (temp >= 2)
      count++;
  }
  cout << count << endl;
  return 0;
}
