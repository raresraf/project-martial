#include <iostream>
using namespace std;

main() {
  int n;
  cin >> n;
  int max = 0, no = 0;
  while (n > 0) {
    int a, b;
    cin >> a >> b;
    no = no + (b - a);
    if (max < no)
      max = no;
    n -= 1;
  }
  cout << max << endl;
}
