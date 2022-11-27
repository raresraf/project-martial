#include <iostream>
using namespace std;
int main() {
  int w;
  cin >> w;
  if ((w % 2 == 1) || (w == 2)) {
    cout << "NO";
  } else {
    cout << "YES";
  }
  return 0;
}
// cout<<(w<3||w%2?"NO":"YES");
