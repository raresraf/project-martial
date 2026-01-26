#include <iostream>
using namespace std;

int main() {
  int n;
  cin >> n;
  int sure[3][n];
  for (int i = 0; i < n; i++)
    cin >> sure[0][i] >> sure[1][i] >> sure[2][i];

  int attempt = 0, c = 0;
  for (int i = 0; i < n; i++) {
    c = sure[0][i] + sure[1][i] + sure[2][i];
    if (c >= 2)
      ++attempt;
    c = 0;
  }

  cout << attempt << endl;

  return 0;
}
