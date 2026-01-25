#include <iostream>

using namespace std;

int main() {
  string hello = "hello";
  int a = 0, c = 0;
  string palabra;
  cin >> palabra;
  for (int i = 0; i < palabra.size(); i++) {
    if (palabra[i] == hello[a]) {
      a++;
      c++;
    }
  }
  (c == 5) ? cout << "YES" : cout << "NO";
  return 0;
}
