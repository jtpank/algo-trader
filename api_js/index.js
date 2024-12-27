const axios = require('axios');
const path = require('path');
const fs = require('fs');

const SEC_API = "https://api.sec-api.io/insider-trading";
const { sec_key } = require("./secrets.json");

async function get_data() {
    const symbol = "CVNA";
    const start_date = "2020-01-01";
    const end_date = "2024-12-26";
    const size = 50;
    let total = 1000;
    for (let from = 50; from < total; from += size) {
        const body = {
            query: `issuer.tradingSymbol:${symbol} AND periodOfReport:[${start_date} TO ${end_date}]`,
            from,
            size,
            sort: [{ filedAt: { order: "asc" } }]
        }
        const {data} = await axios.post(`${SEC_API}?token=${sec_key}`, body);
        fs.writeFileSync(path.join("./cvna_filings", `cvna_filings_${from}_${from+size}.json`), JSON.stringify(data, null, 2))
    }
}

function get_all_transactions() {
    const all_transactions = []
    const size = 50;
    for (let from = 0; from < 1000; from += size) {
        const filestr = fs.readFileSync(path.join("./cvna_filings", `cvna_filings_${from}_${from+size}.json`), { encoding: 'utf-8' });
        const fileobj = JSON.parse(filestr);
        all_transactions.push(...fileobj.transactions);
    }
    return all_transactions;
}

async function main() {
    // const latest_date = new Date('2021-12-01T18:37:53-04:00')
    // const latest_date = new Date('2021-10-24T18:37:53-04:00')
    // const latest_date = new Date('2022-07-03T18:37:53-04:00')
    const all_transactions = get_all_transactions()//.filter(t => (new Date(t.filedAt)) < latest_date);
    const all_people = [];
    for (const transaction of all_transactions) {
        const owner = transaction.reportingOwner.name;
        let person = all_people.find(p => p.name === owner);
        if (!person) {
            person = { name: owner, stocks: [], derivatives: [], history: [] };
            all_people.push(person);
        }
        const report_period = transaction.periodOfReport;

        const ndt = transaction.nonDerivativeTable;
        if (ndt) {
            ndt.transactions = ndt.transactions ?? [];
            // ndt.holdings = ndt.holdings ?? [];
            for (const nd_transaction of ndt.transactions) {
                const owns_directly = nd_transaction.ownershipNature.directOrIndirectOwnership === "D";
                const security_name = nd_transaction.securityTitle;
                let owning_entity = "personally_owned";
                if (!owns_directly) {
                    owning_entity = nd_transaction.ownershipNature.natureOfOwnership;
                }
                let person_stock = person.stocks.find(s => s.name === security_name && s.owning_entity === owning_entity);
                if (!person_stock) {
                    person_stock = { name: security_name, owning_entity, amount: nd_transaction.postTransactionAmounts.sharesOwnedFollowingTransaction };
                    person.stocks.push(person_stock);
                }

                // Just straight update the number of shares owned
                person_stock.amount = nd_transaction.postTransactionAmounts.sharesOwnedFollowingTransaction;
            }
        }

        const dt = transaction.derivativeTable;
        if (dt) {
            dt.transactions = dt.transactions ?? [];
            // dt.holdings = dt.holdings ?? [];
            for (const d_transaction of dt.transactions) {
                const owns_directly = d_transaction.ownershipNature.directOrIndirectOwnership === "D";
                const security_name = d_transaction.securityTitle;
                let owning_entity = "personally_owned";
                if (!owns_directly) {
                    owning_entity = d_transaction.ownershipNature.natureOfOwnership;
                }
                const underlying_security = d_transaction.underlyingSecurity.title;
                let person_deriv = person.derivatives.find(s => s.name === security_name && s.owning_entity === owning_entity && s.underlying_security === underlying_security);
                if (!d_transaction.postTransactionAmounts) console.log(dt.holdings)
                if (!person_deriv) {
                    person_deriv = { name: security_name, owning_entity, underlying_security, amount: d_transaction.postTransactionAmounts.sharesOwnedFollowingTransaction };
                    person.derivatives.push(person_deriv);
                }

                // Just straight update the number of shares owned
                person_deriv.amount = d_transaction.postTransactionAmounts.sharesOwnedFollowingTransaction;
            }
        }
        person.history.push({ date: report_period, stocks: JSON.parse(JSON.stringify(person.stocks)), derivatives: JSON.parse(JSON.stringify(person.derivatives))})
    }

    // const garcia = all_people.find(p => p.name === "GARCIA ERNEST C. II");
    const util = require('util');
    
    const stock_names = all_people.flatMap(person => person.stocks.map(s => `${person.name}_${s.owning_entity}_${s.name}`))
    const deriv_names = all_people.flatMap(person => person.derivatives.map(d => `${person.name}_${d.owning_entity}_${d.name}`))
    const all_columns = ['Date'].concat(stock_names, deriv_names);
    let str = all_columns.join(',') + '\n';
    const all_histories = all_people.flatMap(p => {
        // if (!p.history.stocks) console.log(util.inspect(p, false, null, true)) && process.exit(1)
        p.history.forEach(h => {
            h.stocks.forEach(s => s.person_name = p.name);
            h.derivatives.forEach(d => d.person_name = p.name)
        });
        return p.history
    });
    const histories_no_copies = all_histories.reduce((acc, curr) => {
        if (acc[curr.date]) {
            acc[curr.date].stocks.push(...curr.stocks)
            acc[curr.date].derivatives.push(...curr.derivatives)
        } else {
            acc[curr.date] = {
                stocks: curr.stocks,
                derivatives: curr.derivatives,
            }
        }
        return acc;
    }, {});
    const histories_no_copies_arr = Object.entries(histories_no_copies).map(([key, val]) => ({date: key, ...val}));
    // console.log(histories_no_copies_arr)
    histories_no_copies_arr.sort((a, b) => {
        const [ya, ma, da] = a.date.split('-').map(c => parseInt(c))
        const [yb, mb, db] = b.date.split('-').map(c => parseInt(c))
        return new Date(ya, ma, da) - new Date(yb, mb, db);
    });
    const sorted_histories = histories_no_copies_arr
    // console.log(sorted_histories)
    const acc_history = { stocks: {}, derivatives: {}};
    stock_names.forEach(n => acc_history.stocks[n] = 0);
    deriv_names.forEach(n => acc_history.derivatives[n] = 0);
    for (const history_item of sorted_histories) {
        const { date, stocks, derivatives } = history_item;
        const stock_str_arr = stock_names.map(n => {
            const stock = stocks.find(s => n === (`${s.person_name}_${s.owning_entity}_${s.name}`));
            // if (n === "KEETON RYAN S._personally_owned_Class A Common Stock") console.log(`${stock?.amount}, ${date}: ${acc_history.stocks[n]}`)
            // console.log('stock: ', stock);
            if (!stock) return acc_history.stocks[n];
            acc_history.stocks[n] = stock.amount;
            return stock.amount;
        });
        const deriv_str_arr = deriv_names.map(n => {
            const derivative = derivatives.find(d => n === (`${d.person_name}_${d.owning_entity}_${d.name}`));
            if (!derivative) return acc_history.derivatives[n];
            acc_history.derivatives[n] = derivative.amount;
            return derivative.amount
        });
        const all_vals = [date].concat(stock_str_arr, deriv_str_arr);
        str += all_vals.join(',') + '\n';
    }
    fs.writeFileSync('./people.csv', str);
}

main()