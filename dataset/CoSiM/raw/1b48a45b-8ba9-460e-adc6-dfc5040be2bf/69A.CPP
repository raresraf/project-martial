// codeforces69a.cpp
#include <iostream>
using namespace std;
int main(int argc, char const *argv[]) {
  int n;
  cin >> n;
  int x = 0, y = 0, z = 0;
  int a, b, c;
  while (n--) {
    cin >> a >> b >> c;
    x += a;
    y += b;
    z += c;
  }
  if (x == 0 && y == 0 && z == 0) {
    cout << "YES";
  } else
    cout << "NO";
  return 0;
}