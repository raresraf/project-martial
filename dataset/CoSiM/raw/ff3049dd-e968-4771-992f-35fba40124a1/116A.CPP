#include <iostream>
using namespace std;

int main() {
  int n, intram = 0, max = 0;
  cin >> n;

  int stop[n][2];
  for (int i = 0; i < n; i++)
    for (int j = 0; j < 2; j++)
      cin >> stop[i][j];

  for (int i = 0; i < n; i++) {
    intram -= stop[i][0];
    intram += stop[i][1];
    if (intram > max)
      max = intram;
  }

  cout << max << endl;

  return 0;
}