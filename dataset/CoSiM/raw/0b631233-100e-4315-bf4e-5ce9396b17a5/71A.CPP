#include <iostream>
#include <string.h>
using namespace std;
int main() {
  int n;
  cin >> n;
  cin.ignore();
  string a[n];
  for (int i = 0; i < n; i++) {
    cin >> a[i];
    if (a[i].length() > 10)
      cout << a[i][0] << (a[i].length() - 2) << a[i][(a[i].length()) - 1]
           << endl;
    if (a[i].length() <= 10)
      cout << a[i] << endl;
  }
  return 0;
}
