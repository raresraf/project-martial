#include <iostream>
using namespace std;

int main() {
  int n, count = 0;
  cin >> n;
  int data[n][3];

  for (int i = 0; i < n; i++)
    for (int j = 0; j < 3; j++)
      cin >> data[i][j];

  for (int i = 0; i < n; i++)
    if (data[i][0] + data[i][1] + data[i][2] >= 2)
      count++;

  cout << count << endl;

  return 0;
}