/**
 * I'm sorry I was an Idiot
 */
#include <iostream>
#include <string>
#define toLow(x) (x + 32)

using namespace std;

int main() {
  string a, b;
  cin >> a >> b;
  for (int i = 0; i < a.size(); ++i) {
    char one, two;
    one = a[i];
    two = b[i];
    if (one < 97)
      one = toLow(one);
    if (two < 97)
      two = toLow(two);
    if (one < two) {
      cout << -1 << endl;
      return 0;
    }
    if (one > two) {
      cout << 1 << endl;
      return 0;
    }
  }
  cout << 0 << endl;
  return 0;
}