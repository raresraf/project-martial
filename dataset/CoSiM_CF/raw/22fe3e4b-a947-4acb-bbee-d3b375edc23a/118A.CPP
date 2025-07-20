#include <bits/stdc++.h>
using namespace std;

int main() {
  string line, newLine = "";

  cin >> line;

  for (int i = 0; i < line.length(); i++) {
    if (line[i] == 'a' || line[i] == 'A' || line[i] == 'e' || line[i] == 'E' ||
        line[i] == 'i' || line[i] == 'I' || line[i] == 'o' || line[i] == 'O' ||
        line[i] == 'u' || line[i] == 'U' || line[i] == 'y' || line[i] == 'Y')
      continue;
    else if (isupper(line[i]))
      newLine = newLine + '.' + char(tolower(line[i]));
    else
      newLine = newLine + '.' + line[i];
  }

  cout << newLine << endl;

  return 0;
}

/*
tour
Codeforces
aBAcAba
*/
