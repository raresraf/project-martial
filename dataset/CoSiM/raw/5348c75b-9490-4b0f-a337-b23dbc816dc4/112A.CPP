#include <algorithm>
#include <iostream>
#include <string>

using namespace std;

int main() {
  string a, b;
  cin >> a >> b;

  transform(a.begin(), a.end(), a.begin(), ::toupper);
  transform(b.begin(), b.end(), b.begin(), ::toupper);
  if (a == b) {
    cout << 0 << endl;
  }
  if (a > b) {
    cout << 1 << endl;
  }
  if (a < b) {
    cout << -1 << endl;
  }
  return 0;
}