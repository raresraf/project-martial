#include <iostream>
#include <string>

using namespace std;

int main() {
  int n;
  cin >> n;
  while (n--) {
    string r;
    cin >> r;
    if (r.size() <= 10) {
      cout << r << endl;
      continue;
    }
    cout << r[0] << r.size() - 2 << r[r.size() - 1] << endl;
  }
  return 0;
}