#include <iostream>
#include <string>
using namespace std;
void print_modified(string longword);
int main() {
  int n_words(0), iter(0);
  cin >> n_words;
  string input_words[n_words];
  for (iter = 0; iter < n_words; iter++) {
    cin >> input_words[iter];
  }
  for (iter = 0; iter < n_words; iter++) {
    print_modified(input_words[iter]);
  }
  return 0;
}
void print_modified(string longword) {
  int wordlength(longword.length()), iter(0);
  string output_word;
  if (wordlength > 10) {
    cout << longword[0] << (wordlength - 2) << longword[wordlength - 1] << endl;
  } else {
    cout << longword << endl;
  }
}
