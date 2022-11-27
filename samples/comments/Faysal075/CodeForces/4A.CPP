#include <iostream>
using namespace std;
int main() {
  int w;
  cin >> w;
  string output = (w <= 2 || w % 2 != 0) ? "NO" : "YES";
  cout << output << endl;

  return 0;
}
