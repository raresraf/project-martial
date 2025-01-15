#include <iostream>

using namespace std;

int main() {
  int n, c = 0;
  bool petya, vasya, tonya;
  cin >> n;
  for (int i = 0; i < n; i++) {
    cin >> petya >> vasya >> tonya;
    if (petya + vasya + tonya >= 2)
      c++;
  }

  cout << c;

  return 0;
}
