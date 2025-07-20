#include <cstring>
#include <iostream>
#define size 1000000
using namespace std;
int main() {
  int n, num;
  char word[size], f, l;
  cin >> n;
  while (n--) {
    num = 0;
    cin >> word;
    int length = strlen(word);
    if (length > 10) {
      f = word[0];
      l = word[length - 1];
      cout << f << length - 2 << l << endl;
    } else
      cout << word << endl;
  }

  return 0;
}
