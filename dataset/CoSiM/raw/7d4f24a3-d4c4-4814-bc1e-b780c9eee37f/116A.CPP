#include <iostream>
using namespace std;
int main() {

  int n;
  cin >> n;
  int pass = 0;
  int maxx = 0;

  for (int i = 0; i < n; i++) {
    int sv1, sv2;
    cin >> sv1 >> sv2;
    pass -= sv1;
    pass += sv2;
    pass > maxx ? maxx = pass : maxx;
  }

  cout << maxx;

  return 0;
}
