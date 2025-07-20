#include <iostream>
#include <string>

using namespace std;

int main() {
  string a, b;

  cin >> a >> b;
  for (int i = 0; i < a.size(); ++i) {
    a[i] = tolower(a[i]);
    b[i] = tolower(b[i]);
  }
  if (a < b) {
    cout << -1;
  } else if (a > b) {
    cout << 1;
  } else {
    cout << 0;
  }

  return 0;
}
