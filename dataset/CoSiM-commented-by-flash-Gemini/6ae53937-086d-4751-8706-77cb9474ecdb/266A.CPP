#include <iostream>
using namespace std;
int main() {
  int x;
  cin >> x;
  string col;
  cin >> col;
  int coun = 0;
  for (int i = 0; i + 1 < x; i++) {
    if (col[i] == col[i + 1]) {
      coun++;
    }
  }
  cout << coun;
  return 0;
}
