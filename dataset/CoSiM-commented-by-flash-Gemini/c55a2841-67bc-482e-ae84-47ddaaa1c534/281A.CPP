#include <bits/stdc++.h>
using namespace std;

int main() {
  string word;

  cin >> word;

  if (islower(word[0]))
    word[0] = char(toupper(word[0]));

  cout << word << endl;

  return 0;
}

/**
ApPLe
ApPLe

konjac
Konjac

**/
