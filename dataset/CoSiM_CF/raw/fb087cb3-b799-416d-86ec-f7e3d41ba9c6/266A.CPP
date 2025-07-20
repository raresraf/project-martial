#include <iostream>
#include <string.h>
using namespace std;

int main() {
  int n;
  int count = 0;
  char str[50];
  cin >> n;
  cin >> str;
  for (int i = 0; i < n; i++) {
    if (str[i] == str[i + 1])
      count++;
    else
      continue;
  }
  cout << count << endl;

  return 0;
}
