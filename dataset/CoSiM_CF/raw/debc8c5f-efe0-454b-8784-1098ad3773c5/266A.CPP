#include <iostream>
#include <string>
using namespace std;

int main() {
  int n, pos = 0, cont = 0;
  string word;
  cin >> n;
  cin >> word;
  for (int i = 1; i < n; i++) {
    if (word[i] == word[pos])
      cont++;
    else
      pos = i;
  }
  cout << cont << endl;
  return 0;
}
