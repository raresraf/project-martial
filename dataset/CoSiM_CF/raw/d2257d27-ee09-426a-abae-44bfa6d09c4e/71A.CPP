// 71A - Way Too Long Words

#include <iostream>

using namespace std;

int main() {
  string word[100];
  int n, c, i;

  cin >> n;

  for (i = 0; i < n; i++) {
    cin >> word[i];
    c = word[i].size();

    if (c <= 10)
      cout << word[i] << endl;
    else
      cout << word[i][0] << c - 2 << word[i][c - 1] << endl;
  }

  return 0;
}
