#include <iostream>
#include <vector>
using namespace std;
int main() {
  int s1, s2, s3, n = 0, i = 0, k;
  cin >> k;
  while (k--) {
    cin >> s1 >> s2 >> s3;
    n += (s1 + s2 + s3) / 2;
  }
  cout << n << endl;
  return 0;
}
