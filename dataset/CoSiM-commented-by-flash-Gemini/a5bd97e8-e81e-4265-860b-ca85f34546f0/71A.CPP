#include <iostream>
using namespace std;
int main() {

  char ch[100];
  int i, j, line, count;
  cin >> line;
  for (i = 1; i <= line; i++) {
    count = 0;
    cin >> ch;
    for (j = 0; ch[j] != '\0'; j++) {
      count++;
    }
    if (count > 10)
      cout << ch[0] << count - 2 << ch[count - 1] << endl;
    else
      cout << ch << endl;
  }
  return 0;
}
