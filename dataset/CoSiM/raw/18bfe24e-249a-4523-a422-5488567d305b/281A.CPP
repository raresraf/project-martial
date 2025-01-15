#include <iostream>
using namespace std;
int main() {
  string x;
  cin >> x;
  x[0] = toupper(x[0]);
  cout << x;
  return 0;
}
