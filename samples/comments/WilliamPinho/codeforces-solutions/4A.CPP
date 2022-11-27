#include <iostream>

using namespace std;

int main() {
  int t;
  cin >> t;
  cout << (t % 2 == 0 && t != 2 ? "YES" : "NO");
}