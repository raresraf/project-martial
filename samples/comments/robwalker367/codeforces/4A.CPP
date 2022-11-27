#include <iostream>
using namespace std;

int main() {
  ios_base::sync_with_stdio(false);
  cin.tie(NULL);

  int w;
  cin >> w;
  cout << (w % 2 || w <= 2 ? "NO" : "YES") << "\n";
}