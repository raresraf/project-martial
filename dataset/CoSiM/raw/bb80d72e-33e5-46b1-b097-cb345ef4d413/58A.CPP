#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>

using namespace std;

int main() {
  string A = "hello";
  string B;
  cin >> B;
  int length = B.length();
  int a = 0;
  int i, count = 0;
  for (i = 0; i < length; i++) {
    if (A[i] == B[a]) {
      a++;
      count++;
    }
  }
  if (count == 5)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;

  return 0;
}
